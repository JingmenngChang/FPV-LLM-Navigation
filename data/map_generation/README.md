# Frozen map metadata

This directory contains the retained metadata for the map-generation and selection pipeline.

- `Features_final.csv`: the frozen original-language source table for 10,000 generated maps (84 columns including filename and difficulty rate).
- `Features_final_translated.csv`: the analysis-ready English-column table with filename, difficulty rate, and 84 objective complexity features.
- `difficulty_per_image_final.csv`: frozen map-difficulty estimates.
- `selected_maps_1000.json`: the archived 1,000-map candidate list.
- `final_60_map_ids.csv`: the final 60 map IDs.
- `generation_stats.json`: generation summary metadata.
- `map_selection_terrain_audit_20260806.json`: contemporaneous provenance audit.

The final annotations, start–goal configurations, and trap definitions are under `LlmFPV-main/`. The historical archive does not contain one complete orchestration script that reproduces the 10,000→1,000→60 selection byte for byte; see the repository `DATA.md` for the precise limitation.
