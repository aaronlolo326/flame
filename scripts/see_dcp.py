import argparse
from pathlib import Path

import torch.distributed.checkpoint as DCP


DEFAULT_DCP_PATH = Path("/storage/backup/hei/ttt/flame/seeds/qwen3_tttip_4B/step-0")
DEFAULT_DCP_PATH = Path("/storage/backup/hei/ttt/flame/exp/20260420_tttp_cont1-test/checkpoint/step-1")


def load_state_dict_keys(dcp_path: Path) -> list[str]:
    metadata = DCP.FileSystemReader(str(dcp_path)).read_metadata()
    return list(metadata.state_dict_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DCP checkpoint and print its state dict keys.")
    parser.add_argument("dcp_path", nargs="?", type=Path, default=DEFAULT_DCP_PATH)
    parser.add_argument("--prefix", type=str, default=None, help="Only print keys that start with this prefix.")
    args = parser.parse_args()

    args.dcp_path = "/storage/backup/hei/ttt/flame/exp/20260420_tttp_cont1-test/checkpoint/step-0"
    args.dcp_path = "/storage/backup/hei/ttt/flame/seeds/qwen3_tttip_4B/step-0"
    state_dict_keys = load_state_dict_keys(args.dcp_path)
    if args.prefix is not None:
        state_dict_keys = [key for key in state_dict_keys if key.startswith(args.prefix)]

    print(f"DCP path: {args.dcp_path}")
    print(f"State dict key count: {len(state_dict_keys)}")
    for key in state_dict_keys:
        print(key)


if __name__ == "__main__":
    main()
