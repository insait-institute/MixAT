from baselines import get_method_class
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Merge batched test cases in save_dir into test_cases.json.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    parser.add_argument("--lora_scaling_experiments", action="store_true",
                        help="Used when experimenting with scaling the LORA weights")
    parser.add_argument(
        "--multiple_checkpoints",
        action="store_true",
        help="Whether doing multiple checkpoint experiments or not",
    )
    parser.add_argument(
        "--begin",
        type=int,
        help="Starting of checkpoint",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Ending of checkpoint",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Checkpoint exmination step",
    )
    parser.add_argument(
        "--final",
        type=int,
        help="Final checkpoint number",
    )
    parser.add_argument(
        "--list_num",
        default=None,
        type=str,
        help="Numbers separated with comas. If provided this fill be the interval",
    )
    args = parser.parse_args()

    return args

def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    method_name = args.method_name

    # ========== merge test cases ========== #
    print('Merging test cases')
    method_class = get_method_class(method_name)
    assert (not args.lora_scaling_experiments) or  (not args.multiple_checkpoints), "ERROR"
    if args.lora_scaling_experiments:
        if args.list_num:
            interval = args.list_num.split(",")
            interval = [int(num)/100 for num in interval]
        else:
            interval = [-0.5,0.0,0.5,1.0,1.5]
        print("interval")
        print(interval)
        for i,lam in enumerate(interval):
            new_save_dir = args.save_dir + f"_lambda_{lam}"
            method_class.merge_test_cases(new_save_dir)
    elif args.multiple_checkpoints:
        interval = list(range(args.begin,args.end,args.step))
        interval += [args.final]
        if args.list_num:
            interval = args.list_num.split(",")
            interval = [int(num) for num in interval]
        for i,lam in enumerate(interval):
            new_save_dir = args.save_dir + f"_checkpoint_{lam}"
            method_class.merge_test_cases(new_save_dir)
    else:
        method_class.merge_test_cases(args.save_dir)


if __name__ == "__main__":
    main()
