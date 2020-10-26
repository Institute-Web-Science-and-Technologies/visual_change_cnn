# VISUAL CHANGE WITH CNN

## DESCRIPTION

This is the code I used in my bachelor thesis, which can be found here:

[Link](https://west.uni-koblenz.de/studying/bachelor-theses)
//
[Download](https://west.uni-koblenz.de/assets/theses/bachelor-dvossen.pdf)

No functionality changes were made since, only a small code clean-up.

## UBUNTU

#### INSTALLATION
What you need: 
- the dataset `/Dataset_visual_change/` from <https://zenodo.org/record/3908124>
- extract and integrate the included view masks from `view_masks.zip` in the `/Dataset_visual_change/` data structure

Install following python modules:
- opencv-python
- torch
- colorama

After that run `train_cnn.py help` in your terminal for further instructions.

#### USAGE
##### TRAIN CNN

Run the script with:

`python3 train.py PATH/TO/DATA/ model_name <WEBSITE_NAME> <USER> modi**(optional)`

`<WEBSITE_NAME>` may be any combination of:

`amazon, cnn, gm, guardian, kia, mayo, nih, nissan, reddit, steam, walmart, webmd`

`<USER>` may be any combination of: 

`p1, p2, p3, p4`

Several users and websites are seperated by space, do not use any comma.

Include `help` as parameter to show all available modi.

##### TEST CNN

Run the script with:

`python3 test.py PATH/TO/DATA/ model_name <WEBSITE_NAME> <USER> modi**(optional)`

Parameter work the same as with training.

**WARNING:** Both training and testing may take a very long time!
