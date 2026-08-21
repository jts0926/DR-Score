# Model weights

Model weights are not stored in this GitHub repository. Download both files
from the [DR Score model-weights folder on MEGA](https://mega.nz/folder/b3JGwLxB#TPElfEDOINvw10WS95Ra2w)
and place them in this directory:

```text
knee_detector.pt
drscore_final.pt
```

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `knee_detector.pt` | 165,729,471 | `8414DD151ACE4BDDA83A3218EAF377C3502284BB1E7A64C8079CCAF4F6283A58` |
| `drscore_final.pt` | 53,622,631 | `3B9B382773B64F1057B00DAF5F1C1D1E08226521A46292FA709AA3370973AFE4` |

`knee_detector.pt` contains the Faster R-CNN ResNet-50 FPN localisation state
dictionary. `drscore_final.pt` contains the validation-selected DR Score model
state dictionary. Downloaded `.pt` files are ignored by Git.
