> **NeuroAssist: An Open-Source Tool for**
>
> **Automatic Event Detection in Scalp EEG**
>
> **Authors:** Muhammad Ali Alqarni, Adil Jowad Qureshi, Muiz Alvi,
> Haziq Arbab, Hira Masood, Hassan Aqeel Khan, Awais Mehmood Kamboh,
> Saima Shafait and Faisal Shafait
>
> **Affiliation:** School of Electrical Engineering and Computer
> Science, National University of Sciences and Technology, Islamabad,
> Pakistan
>
> College of Computer Science and Engineering, University of Jeddah,
> Jeddah, Saudi Arabia
>
> Pak-Emirates Military Hospital, Abid Majeed Road Rawalpindi, Pakistan
>
> Deep Learning Laboratory, National Center of Artificial Intelligence,
> Islamabad, Pakistan
>
> **Follow these steps to execute model comparison experiments and
> reproduce reported results:**

1.  **Clone the entire repository**

2.  **Download the dataset using the following link:**

3.  **Place the downloaded dataset along with all the feature files and
    > relevant model files inside the directory of the experiment you
    > want to execute. The code expects these files to be present in the
    > experiment\'s root folder**

4.  **Ensure your execution environment has the following Python
    > dependencies installed and working:**

-   **Python 3.9 or above**

-   **Python MNE library**

-   **Keras and Tensorflow**

-   **Scikit-learn**

> **Please ensure a GPU enabled machine for faster inference and
> training times.**

5.  **Open the terminal inside the directory of the experiment**

6.  **Execute the following commands in order:**

> **\$ python abnormal_img_extractor.py**
>
> **\$ python normal_img_extractor.py**
>
> **Running these code should \[ Insert text here \]**

7.  **Once completed, open the deeplearning_models.ipynb notebook in
    > Jupyter Notebook or VS Code (with Jupyter extension) and run each
    > cell step by step to reproduce our results**

8.  **Once these steps are completed, run the following command:**

> **\$ python window_tracker.py**
>
> **This step is required before using the machine learning notebook**

9.  **Now open the machinelearning_models.ipynb notebook in Jupyter
    > Notebook or VS Code (with Jupyter extension) and run each cell
    > step by step to reproduce our results**

> **Follow these steps to make predictions on an entire EEG waveform:**

1.  **Choose an EEG waveform (in EDF format), along with its
    > corresponding CSV file from the raw_data folder**

2.  **Open the following notebook in Jupyter Notebook or VS Code (with
    > Jupyter extension):**

> **Insert Notebook name here**

3.  **Run each cell step by step**

4.  **When prompted, enter the name of the EEG file you wish to make
    > predictions on. The code will produce a single EEG plot with
    > labels (in the form of rectangles) and predictions (in the form of
    > shading)**

**Note: Use random seed of 444 for reproducing more similar results**

**The following is a table of file descriptions:**

  --------------------------------------------------------------------------------------
  **File Name**                              **Description**
  ------------------------------------------ -------------------------------------------
  **raw_data**                               Contains EDF and CSV files. The EDF files
                                             are EEG waveforms in European Data Format,
                                             while the CSV files contain timestamps
                                             corresponding to the abnormalities present
                                             in corresponding EDF files. EDF files have
                                             been categorized into Normal and Abnormal
                                             subfolders

  **utils.py**                               Contains code for functions needed to
                                             extract windows and labels from raw data.
                                             EDF files are broken into 50% overlapping
                                             windows of size 400 samples. This size
                                             corresponds to 2 second as the sampling
                                             frequency of the EDF files were 200 Hz

  **abnormal_img_extractor.py**              Transforms windows into cwt scalograms
                                             corresponding to each of the label classes.
                                             A window is considered abnormal if there is
                                             at least 25% abnormality present in it

  **normal_img_extractor.py**                Produces random normal images from all of
                                             the normal edf files. Normal class
                                             scalograms are undersampled to around 432k
                                             images by randomly taking 445 windows from
                                             each normal EDF file

  **deeplearning_cwt_images.zip**            Contains cwt scalogram images randomly
                                             separated into train, validation and test
                                             folders. Images have been separated into
                                             70% train, 15% validation and 15% test
                                             sets. These data sets are used in the deep
                                             learning models. These folders are created
                                             by code in the deeplearning_models.ipynb
                                             notebook

  **deeplearning_models.ipynb**              Contains deep learning models, namely vgg16
                                             and Googlenet. This code processes the cwt
                                             image scalograms, passes them through the
                                             models and produces deep learning results,
                                             classifying each window

  **window_tracker.py**                      Traces the original windows that are
                                             transformed into cwt scalograms placed in
                                             **deeplearning_cwt_images.zip** and saves
                                             the corresponding EDF file name, channel
                                             number, window number and label into a
                                             **window_tracker.txt file**

  **window_tracker.txt**                     This File is a record of every single
                                             window being used in the cwt images data
                                             set. This file is then used to reproduce
                                             the same training and test sets for the
                                             machine learning models. Since machine
                                             learning models are not utilizing a
                                             validation dataset, validation data has
                                             been added to the training set and we now
                                             have a 85% training and 15% test data
                                             split. Both machine learning dwt
                                             statistical feature data points and deep
                                             learning model cwt image data test sets
                                             have been produced from the same windows
                                             using this file.

  **machinelearning_dwt_stats_arrays.zip**   Contains numpy files corresponding to the
                                             processed training data arrays, training
                                             label arrays, test data arrays and test
                                             label arrays. These arrays have been
                                             produced by the
                                             machinelearning_models.ipynb notebook

  **machinelearning_models.ipynb**           Contains machine learning models, namely a
                                             decision tree classifier, random forest
                                             classifier and support vector classifier.
                                             This code selects windows mentioned in the
                                             **window_tracker.txt** file and extracts
                                             them from EDF files using utils.py. It is
                                             important to state the position of the
                                             'train', 'validation' and 'test' keywords
                                             in the window_tracker.txt file in the
                                             train_pos, validation_pos and test_pos
                                             variables in this notebook. The windows are
                                             transformed into dwts of 1 to 4 scales
                                             using 'db2' as the mother wavelet.
                                             Statistical features including mean,
                                             maximum value, minimum value, standard
                                             deviation and variance are extracted from
                                             these data points. We then apply PCA of
                                             dimensionality of 2 to these features and
                                             split them into numpy arrays of 85%
                                             training and 15% test split. These data
                                             sets are then passed through the models to
                                             produce results of classification of
                                             windows
  --------------------------------------------------------------------------------------
