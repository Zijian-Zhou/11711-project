# this does not work on fairseq 0.10
# for fairseq 0.9.0, you will need to edit the call_main function in the distributed_utils.py file
# replace main(args, kwargs) with main(args, **kwargs) in the single GPU main case
#
# you will also need to edit the main function in eval_lm.py
# comment out "args.tokens_per_sample -= args.context_window"
# comment out the if section starting with "if args.add_bos_token:"

python fairseq_cli/eval_lm.py data-bin/iwslt14.tokenized.de-en \
--path [path/to/checkpoint.pt] \
--task translation --max-sentences 8 \
--source-lang de --target-lang en