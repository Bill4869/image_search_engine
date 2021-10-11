# image_search_engine
This is an original image search engine which includes the below functions.
1. Search
    - Color
    - Shape
    - Object
    - Text
    - Similar images
1. Sort
    - Color
    - Shape
    - Object
    - Text
## Requirements
- numpy
- opencv
- tensorflow
- lpips
- torch
- tensorflow_hub
- easyocr

## Usage
### Search function
#### Search color
```
python .\search_engine.py --search -c 緑
```
#### Search shape
```
python .\search_engine.py --search -s 三角形
```
#### Search object
See `label.csv` for detectable objects
```
python .\search_engine.py --search -o dog
```
#### Search text
Edit variable `SEARCH_LANG` in `search.py` to change detectable languages
```
python .\search_engine.py --search -t [text]
```
#### Search similar images
```
python .\search_engine.py --search -i [target image]
```
With deep features (VGG16)
```
python .\search_engine.py --search -i [target image] --deep
```
You can also find similar images based on one of the following option only. See `find()` in `search_engine.py`
- FID
- LPIPS
- Difference of color histogram
- Deep features
#### Search multiples
```
python .\search_engine.py --search -c [color] -s [shape] -o [object] -t [text] -i [target image] --deep
```

### Sort function
```
python .\search_engine.py --sort -b color
```

**For more options see `search_engine.py`**
