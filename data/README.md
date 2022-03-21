
## Organization of the datasets
 The Datasets should be organized in the following way:

```
data/
├───data_data_CB/	            # Cornell-Box Dataset
│   ├───*_renders_scene_*/      # folders for individual scenes
│   ├───...			
│   ├───train.txt	
│   ├───test.txt	
│   └───val.txt		
|	
├───data_agresti/               # Datasets from Agresti et al.
│   ├───S1/			
│   │   └───synthetic_dataset/			
│   │       ├───test_set/		
│   │       │   └───ground_truth/		
│   │       └───training_set/	
│   │           └───ground_truth/
│   ├───S2/			
│   │   └───S2/	
│   ├───S3/			
│   │   └───S3/			
│   │       └───ground_truth/
│   ├───S4/			
│   │   └───real_dataset/
│   │       └───ground_truth/
│   └───S5/		
│      └───S5/
│           └───ground_truth/
|
└───data_FLAT/                     # FLAT Dataset
    ├───kinect/			
    │   ├───full/			
    │   ├───gt/			
    │   ├───list/			
    │   └───msk/		
    ├───phasor/	    # not used in the code
    └───deeptof/    # not used in the code			
```

Other paths/path_structures may be specified in the respective `data_loader.py` files.