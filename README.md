# Seg_ReID 
All weights are [Here](https://drive.google.com/open?id=1RP2SYc1339XxS_aM3GkPg4cbvGcVpvJr).

## Market1501
| Model                       | mAP |CMC-1 | CMC-3 | CMC-5 | CMC-10 |
| :-------------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | 
| Resnet(crossentropy)|   66.47 | 84.38 | 91.38 | 94.03 | 96.11
| Resnet(triplet+crossentropy)|   78.08  | 90.41 | 94.86 | 96.17 | 97.45 |


## Vivalab's dataset
| Model                       | mAP |CMC-1 | CMC-3 | CMC-5 | CMC-10 |
| :-------------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | 
| CGN(the paper's method)(bounding box)|   69.43 | 83.75 | 87.19 | 88.84 | 91.32 |
| CGN(the paper's method)(mask)|  67.14 | 84.30|  88.77|  90.29|  91.87
| MGN(bouding box)            | 81.24 | 89.92 | 92.58 | 93.23 | 94.96 |
| MGN(mask)                   | 75.04 | 86.85|  89.94|  91.05|  93.04 |
