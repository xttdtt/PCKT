# Related paper

This is the code corresponding to the paper: **PCKT: Problem Composition in Knowledge Tracking**

This is the corresponding paper link:https://dl.acm.org/doi/abs/10.1145/3572549.3572620

If you need to cite this paper, please use one of the following three citation methods:

---

**ACM Ref**

Huaxiong Yao, Juntao Yang, Zuoquan Xie, Jia Guo, Renyi Chen, and Mengling Hu. 2023. PCKT: Problem Composition in Knowledge Tracking. In Proceedings of the 14th International Conference on Education Technology and Computers (ICETC '22). Association for Computing Machinery, New York, NY, USA, 442–448. https://doi.org/10.1145/3572549.3572620

---

**GB/T**

Yao H, Yang J, Xie Z, et al. PCKT: Problem Composition in Knowledge Tracking[C]//Proceedings of the 14th International Conference on Education Technology and Computers. 2022: 442-448.

---

**BibTex**

@inproceedings{yao2022pckt,
  title={PCKT: Problem Composition in Knowledge Tracking},
  author={Yao, Huaxiong and Yang, Juntao and Xie, Zuoquan and Guo, Jia and Chen, Renyi and Hu, Mengling},
  booktitle={Proceedings of the 14th International Conference on Education Technology and Computers},
  pages={442--448},
  year={2022}
}

---

# Code notes

Here is the whole process of running the code, please run the code in this order:

1. you should download the dataset and put it into the *data* folder in the corresponding dataset folder. 

   Take dataset *Assist09* as an example:

   Firstly, you need to first create a folder named *Assist09*

   Secondly, create *data* folder inside the *Assist09* folder

   Finally, rename the downloaded dataset to *Assist09_original.csv* and put it into the *data* folder.

   You can download the dataset here:

   *Assist09*: https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view

   *Assist12*: https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view

2. Set non-fixed parameters in *HyperParameter.py*.

3. After setting the non-fixed parameters, you should run the program in the following order, please be sure to run in this order, otherwise errors will be reported: 

   *ProcessData.py*---->*TrainEmbedding.py*---->*JointEmbedding.py*---->*TrainModel.py*

4. The **best acc** and **best auc** in the *TrainModel.py* are the final running results, representing best accuracy and best ROC curve area respectively.

# Packages used

  You can use the following command to install the package:

   ```shell
pip lnstall tensorflow==1.15.0 pandas numpy scipy sklearn
   ```

   The version of python we are using is 3.7.12

   Please do not run this program in the environment of tensorflow 2.0 or higher, we strongly recommend that you use tensorflow 1.15.0 version to run this model. Other packages can use the latest version.

# Contact information

If you have any questions, please feel free to contact us.

Our email address is 761744650@qq.com and yjt761744650@gmail.com

