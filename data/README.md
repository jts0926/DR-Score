# Private data boundary

No radiographs, outcomes, participant identifiers, or source-cohort extracts are
distributed with this repository. Build `data/metadata.csv` locally using the
schema in `examples/metadata.example.csv`.

The four allowed cohort labels are `OAI`, `MOST`, `KICK`, and `MenTOR`. The
`participant_id` and `knee_id` values should be study-specific pseudonyms. Do not
use names, medical-record numbers, dates of birth, or other direct identifiers.

Image paths can be absolute or relative to the metadata file. All listed images
must already be unilateral, quality-controlled knee crops when training the DR
Score model. The bilateral detector is used by the single-image inference entry
point.
