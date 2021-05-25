[license]: https://github.com/cakuba/Deep-Multimodal-Learning-for-Lymph-Node-Metastasis-Prediction-of-Primary-Thyroid-Cancer/blob/main/LICENSE


# Deep-Multimodal-Learning-for-Lymph-Node-Metastasis-Prediction-of-Primary-Thyroid-Cancer

Incidence of primary thyroid cancer rises steadily over the past decades because of overdiagnosis and overtreatment through the improvement in imaging techniques for screening, especially in ultrasound examination. Metastatic status of lymph nodes is important for staging the type of primary thyroid cancer. Deep learning algorithms based on ultrasound images were thus developed to assist radiologists on the diagnosis of lymph node metastasis. To integrate more clinical context (e.g., health records and various image modalities) into, and explore more interpretable patterns discovered by, these algorithms, a deep multimodal learning network was proposed for the prediction of lymph node metastasis in primary thyroid cancer patients. 

The proposed network is also named as MMC-net (MultiModal classification network).

## Quickstart

0. fully tested on Ubunti 16.04 LTS with cuda 10.1 & cudnn 7.6.5 (Nvidia 3090 GPUs)

1. developed in Python 3.6 with libraries as
```Bash
   Tensorflow 2.3.1
   Keras 2.4
```

2. Step 1 - to train SMC-net (single modality classification network) for each data modality
```Bash
   python3 sm_clinic_classification.py
   python3 sm_color_classification
   python3 sm_color_classification
```
3. Step 2 - to train MMC-net using the trained weights from Step 1
```Bash
   python mm_classification.py 
``` 

## Who are we?

MMC-net is proposed and maintained by researchers from <a href="https://www.wit.edu.cn/" target="_blank">WIT</a>.

## License

See [LICENSE][license]
