# EpiGeoNet

EpiGeoNet is an Explainable Spatio-Temporal Graph Attention Network for district-level disease outbreak early warning using GeoAI. It dynamically fuses weather, satellite LAI, population density, and historical case counts on a dynamically-weighted district graph to forecast cases 1/2/4 weeks ahead. It includes a 4-class risk head and a binary outbreak-onset alert head, with an explainability module that emits human-readable outbreak-risk bulletins.

## Reproduce

To reproduce the entire pipeline and results, run:
\`\`\`bash
make reproduce
\`\`\`
Alternatively, you can run the reproduction script directly:
\`\`\`bash
python scripts/reproduce_all.py
\`\`\`
