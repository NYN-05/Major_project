source_path = r"d:/Implementation/tools/face_roi_pipeline.py"
namespace = {"__name__": "face_module", "__file__": source_path}
with open(source_path, "r", encoding="utf-8") as handle:
	code = compile(handle.read(), source_path, "exec")
exec(code, namespace)
print(sorted(namespace.keys())[:20])
print(sorted(name for name in namespace if name in {"build_parser", "run_pipeline", "discover_frame_sequences", "require_dependencies"}))
parser = namespace["build_parser"]()
args = parser.parse_args(["run", "--videos", "id0_0000", "--workers", "1"])
print(args.command, args.videos, args.workers)
result = namespace["run_pipeline"](args)
print("RESULT", result)
