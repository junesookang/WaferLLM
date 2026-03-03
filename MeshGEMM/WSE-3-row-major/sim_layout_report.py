from cerebras.sdk.layout import SdkLayout # type: ignore

# analyze the compiled binary produced by cslc
layout = SdkLayout("out")

# Generate a static performance report
layout.report("layout_summary.json")

print("✅ Generated layout_summary.json (contains per-region latency and routing estimates)")
