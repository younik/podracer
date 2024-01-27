import optax


def get(args):
    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    return optax.MultiSteps(
                optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
                    ),
                ),
                every_k_schedule=args.gradient_accumulation_steps,
            )