import tensorflow as tf
import numpy as np
import os
import re
import configs
from tensorflow.keras import mixed_precision
from model import is_legacy_checkpoint, load_checkpoint_weights

try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    AdamW = None


class RewarmSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scales a base schedule by a linear ramp that starts at resume_step (used when Adam's state is lost)."""

    def __init__(self, base, resume_step, rewarm_steps):
        self.base = base
        self.resume_step = resume_step
        self.rewarm_steps = rewarm_steps

    def __call__(self, step):
        lr = self.base(step)
        if self.rewarm_steps <= 0:
            return lr
        ramp = (tf.cast(step, tf.float32) - self.resume_step + 1.0) / self.rewarm_steps
        return lr * tf.clip_by_value(ramp, 0.0, 1.0)

    def get_config(self):
        return {"base": self.base, "resume_step": self.resume_step, "rewarm_steps": self.rewarm_steps}


class Trainer():

    def __init__(self, llm, training_generator, val_generator, checkpoint=None, source_layers=None):

        mixed_precision.set_global_policy('mixed_float16')
        self.llm = llm
        self.training_generator = training_generator
        self.val_generator = val_generator

        # linear warmup to the peak LR, then cosine decay to MIN_LR_RATIO * peak
        total_steps = len(self.training_generator)
        warmup_steps = min(configs.WARMUP_STEPS, total_steps // 10)
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                                            initial_learning_rate=0.0,
                                            warmup_target=configs.INITIAL_LEARNING_RATE,
                                            warmup_steps=warmup_steps,
                                            decay_steps=total_steps - warmup_steps,
                                            alpha=configs.MIN_LR_RATIO)

        # checkpoints are named step_<N>.weights.h5 with optimizer state in step_<N>.optimizer.npz
        self.start_step = 0
        self.opt_state_file = None
        self.legacy_checkpoint = False
        if checkpoint:
            match = re.search(r"step_(\d+)", os.path.basename(checkpoint))
            if match:
                self.start_step = int(match.group(1))
            self.legacy_checkpoint = is_legacy_checkpoint(checkpoint)
            candidate = checkpoint.replace(".weights.h5", ".optimizer.npz")
            # Adam moments only fit the architecture they were saved from
            if os.path.exists(candidate) and not source_layers and not self.legacy_checkpoint:
                self.opt_state_file = candidate
            if self.legacy_checkpoint:
                print("Checkpoint predates final_norm; discarding optimizer state")

        # without saved Adam moments the first updates are ~lr per weight, which wrecks the model, so re-warm the LR
        if checkpoint and self.opt_state_file is None:
            print(f"No optimizer state found for {checkpoint}: re-warming the learning rate over {configs.RESUME_REWARM_STEPS} steps")
            self.lr_schedule = RewarmSchedule(self.lr_schedule, self.start_step, configs.RESUME_REWARM_STEPS)

        if AdamW is not None:
            optimizer = AdamW(learning_rate=self.lr_schedule, weight_decay=1e-2)
        else:
            print("tensorflow.keras.optimizers.AdamW is unavailable; falling back to Adam without decoupled weight decay")
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.optimizer = optimizer
        self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # build the model to avoid lazy build during training
        dummy_input = tf.zeros((1, configs.MAX_SEQ_LENGTH), dtype=tf.int32)
        _ = self.llm(dummy_input)
        self.llm.summary()

        # build now so the loss scale variable exists before micro_batch_step is traced
        self.optimizer.build(self.llm.trainable_variables)

        if checkpoint and source_layers:
            self.transfer_weights(checkpoint, source_layers)
            self.optimizer.inner_optimizer.iterations.assign(self.start_step)
            print(f"Transferred weights of the first {source_layers} blocks from {checkpoint}; "
                  f"remaining blocks are freshly initialized. Continuing at step {self.start_step}")
        elif checkpoint:
            load_checkpoint_weights(self.llm, checkpoint)
            self.optimizer.inner_optimizer.iterations.assign(self.start_step)
            if self.opt_state_file:
                self.load_optimizer_state(self.opt_state_file)
            print(f"Resumed from {checkpoint} at step {self.start_step}")
    

    def transfer_weights(self, checkpoint, source_layers):
        # load the checkpoint into a model of its original depth, then copy what it shares with the deeper model
        from model import LLM
        # the main model was created before the mixed precision policy was set, so build the source the same way
        policy = mixed_precision.global_policy()
        mixed_precision.set_global_policy('float32')
        src = LLM(self.llm.embedding.input_dim, self.llm.model_dim, configs.NUM_ATTENTION_HEADS, source_layers,
                  use_tying_embedding=self.llm.use_tying_embedding)
        mixed_precision.set_global_policy(policy)
        src(tf.zeros((1, configs.MAX_SEQ_LENGTH), dtype=tf.int32))
        load_checkpoint_weights(src, checkpoint)

        self.llm.embedding.set_weights(src.embedding.get_weights())
        if self.llm.use_tying_embedding:
            self.llm.lm_bias.assign(src.lm_bias)
        else:
            self.llm.head.set_weights(src.head.get_weights())
        for dst_block, src_block in zip(self.llm.transformers[:source_layers], src.transformers):
            dst_block.set_weights(src_block.get_weights())
        del src

    def save_optimizer_state(self, path):
        state = {f"inner_{i}": v.numpy() for i, v in enumerate(self.optimizer.inner_optimizer.variables)}
        state["loss_scale"] = self.optimizer.dynamic_scale.numpy()
        np.savez(path, **state)

    def load_optimizer_state(self, path):
        state = np.load(path)
        variables = self.optimizer.inner_optimizer.variables
        if len(state.files) - 1 != len(variables):
            raise ValueError(f"Optimizer state in {path} does not match the current model/optimizer")
        for i, v in enumerate(variables):
            v.assign(state[f"inner_{i}"])
        self.optimizer.dynamic_scale.assign(state["loss_scale"])
        print(f"Restored optimizer state from {path}")

    ''''
    @tf.function(jit_compile=False)
    def train(self):

        train_iterator = iter(self.training_generator)
        val_iterator = iter(self.val_generator)
            
        for step in range(len(self.training_generator)):
               

            #print("Training Sample X:", self.training_generator.tokenizer.decode(training_batch['X'][0]))
            #print("Training Sample Y:", self.training_generator.tokenizer.decode(training_batch['Y'][0]))

            #print(training_batch['X'].shape)
            #print(training_batch['Y'].shape)

            accum_steps = configs.BATCH_SIZE // configs.MICRO_BATCH_SIZE
            accum_grads = [tf.zeros_like(v, dtype=tf.float32) for v in self.llm.trainable_variables]

            batch_loss = 0

            for _ in range(accum_steps):

                training_batch = next(train_iterator)

                with tf.GradientTape() as gt:
                    logits = self.llm(training_batch['X'])
                    loss = self.cce(training_batch['Y'], logits)

                    #mask = tf.cast(tf.not_equal(training_batch['Y'], self.training_generator.tokenizer.token_to_id("[PAD]")), tf.float32)
                    #mask = tf.cast(tf.not_equal(training_batch['Y'], self.training_generator.tokenizer.pad_token_id), tf.float32)
                    #micro_batch_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
                    micro_batch_loss = tf.reduce_mean(loss)
                    scaled_loss = self.optimizer.get_scaled_loss(micro_batch_loss) # loss for mixed precision stability

                grads = gt.gradient(scaled_loss, self.llm.trainable_variables)

                for i, g in enumerate(grads):
                    if g is not None:
                        accum_grads[i] += tf.cast(g, tf.float32)

                batch_loss += micro_batch_loss / accum_steps

            # average gradients accross micro batches
            accum_grads = [g / accum_steps for g in accum_grads]
            unscaled_grads = self.optimizer.get_unscaled_gradients(accum_grads)

            # apply gradients
            clipped_grads, _ = tf.clip_by_global_norm(unscaled_grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, self.llm.trainable_variables))

            #memory_info = tf.config.experimental.get_memory_info('GPU:0')
            #print(f"Current: {memory_info['current'] / 1e6:.2f} MB")
            #print(f"Peak: {memory_info['peak'] / 1e6:.2f} MB")

            if step % 10 == 0:

                # To increase validation batch size, no need for the gpu
                with tf.device('/device:CPU:0'):

                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(self.val_generator)
                        val_batch = next(val_iterator)

                    #print(val_batch['X'].shape)
                    #print(val_batch['Y'].shape)
                    #print("Validation Sample X:", self.val_generator.tokenizer.decode(val_batch['X'][0]))
                    #print("Validation Sample Y:", self.val_generator.tokenizer.decode(val_batch['Y'][0]))

                    #logits_val = self.llm(val_batch['X'])
                    #val_loss = self.cce(val_batch['Y'], logits_val)

                    #val_mask = tf.cast(tf.not_equal(val_batch['Y'], self.val_generator.tokenizer.token_to_id("[PAD]")), dtype=tf.float32)
                    #val_batch_loss = tf.reduce_sum(val_loss * val_mask) / tf.reduce_sum(val_mask)
                    #val_batch_loss = tf.reduce_mean(val_loss)

                    #t_loss = batch_loss.numpy()
                    #v_loss = val_batch_loss.numpy()
                    #ppl = np.exp(v_loss)

                    #print(f"step {step}: training_loss = {batch_loss:.4f}, val_loss = {val_batch_loss:.4f}, perpelxity = {tf.exp(val_batch_loss):.4f}")
                    #print(f"step {step}: training_loss = {t_loss:.4f}, val_loss = {v_loss:.4f}, perpelxity = {ppl:.4f}")
                    print(f"step: {step}")

            # save weights every 500 steps
            if ((step + 1) % 2000) == 0:
                self.llm.save_weights(f"checkpoints/step_{step + 1}/llm")
                print(f"Saved parameters at step {step + 1}")

            #self.training_generator.on_epoch_end()
            #self.val_generator.on_epoch_end()
    '''

    

    @tf.function
    def micro_batch_step(self, x, y):

        with tf.GradientTape() as gt:
            logits = self.llm(x, training=True)
            # Loss in float32 for mixed-precision stability
            loss = self.cce(y, tf.cast(logits, tf.float32))
            scaled_loss = self.optimizer.scale_loss(loss)

        # Calculate gradients for this micro-batch only
        grads = gt.gradient(scaled_loss, self.llm.trainable_variables)
        return loss, grads


    def mini_batch_step(self, train_iterator, accum_steps):

        accum_grads = [tf.zeros_like(v) for v in self.llm.trainable_variables]
        total_loss = 0.0

        for _ in range(accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.training_generator)
                batch = next(train_iterator)

            loss, grads = self.micro_batch_step(batch['X'], batch['Y'])
            
            # Accumulate gradients
            for i, g in enumerate(grads):
                if g is not None:
                    accum_grads[i] += g
            total_loss += loss.numpy()

        # Keras 3 LossScaleOptimizer unscales inside apply_gradients (and skips non-finite steps),
        # so keep grads scaled and clip against the scaled norm to get an effective max norm of 1.0
        avg_grads = [g / accum_steps for g in accum_grads]
        clipped_grads, _ = tf.clip_by_global_norm(avg_grads, 1.0 * self.optimizer.dynamic_scale)
        
        self.optimizer.apply_gradients(zip(clipped_grads, self.llm.trainable_variables))
        
        return total_loss / accum_steps, train_iterator

    def train(self):

        train_iterator = iter(self.training_generator)
        val_iterator = iter(self.val_generator)
        
        accum_steps = configs.BATCH_SIZE // configs.MICRO_BATCH_SIZE

        for step in range(self.start_step, len(self.training_generator)):

            batch_loss, train_iterator = self.mini_batch_step(train_iterator, accum_steps)

            # validation and logging every 100 steps
            if step % 50 == 0:

                with tf.device('/device:CPU:0'):

                    # average over several validation batches for a stable estimate
                    val_losses = []
                    for _ in range(configs.NUM_VAL_BATCHES):
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(self.val_generator)
                            val_batch = next(val_iterator)

                        logits_val = self.llm(val_batch['X'])
                        val_losses.append(float(self.cce(val_batch['Y'], logits_val)))

                    val_loss = float(np.mean(val_losses))
                    perplexity = float(np.exp(val_loss))

                print(f"step: {step} | loss: {batch_loss:.4f} | val_loss: {val_loss:.4f} | perplexity: {perplexity:.4f}")

            if (step + 1) % 500 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                self.llm.save_weights(f"checkpoints/step_{step + 1}.weights.h5")
                self.save_optimizer_state(f"checkpoints/step_{step + 1}.optimizer.npz")
                print(f"Saved parameters and optimizer state at step {step + 1}")