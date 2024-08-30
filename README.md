# CTP2Perf2Lesion

This repository contains the code and pre-trained models developed for the [ISLES'24](https://isles-24.grand-challenge.org/) Challenge, where we employed a 2-step inference approach to address the challenges of ischemic lesion detection from CTP source images. 
Predicting lesions directly from CTP images is challenging and often fails to generalize well in testing, necessitating a more intuitive network design.
To overcome this, we focused on key hemodynamic parameters like CBF and TMAX, which are essential for lesion classification. 
We developed a regression model capable of extracting these parameters and lesion areas from CTP source images, effectively capturing meaningful and clear hemodynamic information from the 4D data. 
In the second step, a UNet-based segmentation model generates the ischemic lesion mask from on the HPM regression results. 
This structured approach leverages the relationships between hemodynamic parameters and lesion characteristics, potentially enhancing detection accuracy.



## Installation

* Clone this repository:
   ```
   $ git clone https://github.com/jiwoosong/CTP2Perf2Lesion.git
   ```

* Install the dependencies:
   ``` bash
   $ pip install -r requirements.txt
   ```

## Inference

* Run `inference.py`: you need to change  `INPUT_PATH`, `OUTPUT_PATH`, `RESOURCE_PATH` as necessary path. If you want to see visualization, set `save_suppl=True`.
   ```python
   INPUT_PATH = Path("./test/input")
   OUTPUT_PATH = Path("./test/output")
   RESOURCE_PATH = Path("./resources")
   ```
   ```python
   if __name__ == "__main__":
       raise SystemExit(run(save_suppl=True))
   ```
  
## Results

* supplementary Results
  <img src="/test/output/images/supplementary/0030.png">
  <img src="/test/output/images/supplementary/0040.png">
