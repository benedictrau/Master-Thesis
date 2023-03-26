<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
-->

[![MIT License][license-shield]][license-url]

<h2 align="center">Evaluation of the Use of Reinforcement Learning in Retail Inventory Management</h2>

  <p align="center">
    Master's thesis submitted on 27.03.2023 by Benedict Rau at the PSCM Department of TU Darmstadt

  </p>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#required-packages">Required Packages</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#Overview-of-the-Python-scripts-used">Overview of the Python scripts used</a></li>
      <ul>
        <li><a href="#Predict_Stock">PredictStock</a></li>
      </ul>
      <ul>
        <li><a href="#Simulation_and_Training">Simulation_and_Training</a></li>
      </ul>
      <ul>
        <li><a href="#SimulationStudy">SimulationStudy</a></li>
      </ul>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In the context of the master thesis, the use of reinforcement learning in inventory management in the retail sector was evaluated. <br>
This README file is intended to give an overview of the scripts used in the project and to provide the reader with instructions on how to execute the scripts independently.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Required Packages

The following libraries with their respective versions are used in this project:

* [Pytorch](https://pytorch.org): v1.12.1
* [SimPy](https://simpy.readthedocs.io/en/latest/index.html): v4.0.1
* [NumPy](https://numpy.org): v1.22.4
* [pandas](https://pandas.pydata.org): v1.3.2
* [scikit-learn](https://scikit-learn.org/stable/): v1.2.0
* [Joblib](https://joblib.readthedocs.io/en/latest/): v1.2.0
* [matplotlib](https://matplotlib.org): v3.4.1
* [xgboost](https://xgboost.ai): v1.6.2


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/benedictrau/Master-Thesis.git
   ```
2. Install the required packages
   ```sh
   pip install "package name"
   ```
   If you have already installed the package you can get the required version as follows.
   ```sh
   pip install --force-reinstall -v "package name==1.2.2"
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Overview -->
## Overview of the Python scripts used

In the following, an overview of the scripts used in the project is given based on the folder structure. 

### Predict_Stock

#### Generate the data to train the classification models
The folder _**"Predict_Stock"**_ contains all scripts and files used to generate the belief state. 
The file _**"CreateCSV.py"**_ was used to generate the data for training the classification models.
This script interacts with the reduced simulation model of the inventory system that can be found under the file name _**"Env_predict_stock.py"**_. <br>
The generated .csv files are stored in the folder _**"Data"**_.
<br><br>

#### Classification models
In total, the following three classification algorithms were tested as part of the master's thesis:
* Random forest (_**"RF.py"**_)
* Support vector machine (_**"SVM.py"**_)
* XGBoost (_**"XGB.py"**_)
<br>

The scripts each contain the following four functions:
* _train()_: This function is used to train and save the model using the .csv files from the folder _**"Data"**_.
* _HP_opt()_: This function performs random search to optimize the hyperparameter.
* _predict()_: This function returns the most probable class based on a trained model.
* _class_probability()_: This function returns the probabilities of each class based on a trained model.

The trained models are stored in the _**"Results"**_ folder and are also called up from there for prediction.
<br><br>

### Simulation_and_Training
The folder _**"Simulation_and_Learn"**_ contains all files that are related with the simulation of the inventory system and the training of the Reinforcement Learning agents.

#### Determine_optimal_audit_frequency
This sub-folder contains the scripts required to determine the optimal audit frequency for the EOQ and the (Q,R) policy. The two following scripts are used to derive the optimal audit frequency:
* _**"EOQ_CI_Preselect_Audit_Frequency.py"**_: This script is used to determine the optimal audit frequency for the EOQ policy.
* _**"QR_CI_Preselect_Audit_Frequency.py"**_: This script is used to determine the optimal audit frequency for the (Q,R) policy.

The results from the two evaluations are stored as an .xlsx file in the folder **"Results_Excel"**.
<br><br>

#### Reinforcement_Learning
The following scripts are stored in the folder "Reinforcement_Learning":
* _**MLS.py**_: In this script a Reinforcement Learning agent can be trained using the POMDP approximator "Most Likely State".
* _**QMDP.py**_: In this script a Reinforcement Learning agent can be trained using the POMDP approximator "QMDP".
* _**DMC.py**_: In this script a Reinforcement Learning agent can be trained using the POMDP approximator "Dual Mode Control".

The three scripts are able to train an agent for a given hyperparameter-set. The neural network is saved after training in the folder _**"results_NN"**_.

###### Testing_Hyperparameter
The sub-folder **Testing_Hyperparameter** contains all scripts used to conduct the hyperparameter testing. Thus, three more sub-folders are used for each POMDP approximator. Each of these folders contains the following three scripts:
* **"FactorScreening_**_POMDP Approximator Name_**.py"**: This script is used to conduct the factor screening to determine the hyperparameters that influence the reward most.
* **"RS_**_POMDP Approximator Name_**.py"**: After the hyperparameters are determined that influence the results most, they are used for random search to optimize them. 
* _**"POMDP Approximator Name.py"**_: This script is used for training the agents. The script is called while factor screening and random search.
<br><br>

#### SimulationStudy
In the folder _**"SimulationStudy"**_ all scripts required to conduct the simulation study are stored. The script _**"SimulationStudy.py"**_ is mainly responsible to conduct the simulation study, determine the transient times and calculate the number of simulation runs.
* _det_transient_time()_: The function _det_transient_time_ is used to determine the transient time. Thus, the function creates the plot required for the graphical analysis.
* _confidence_interval()_: This function is used to conduct the simulation study and plot the confidence intervals to compare the different policies.
* _prerun()_: The function _prerun_ is used to determine the number of replications.
<br>

To get determine the actions to be taken according to the policies, three additional scripts exist:
* _**"EOQ.py"**_: This script is used to determine the optimal replenishment quantity following the EOQ model.
* _**"QR.py"**_: This script is used to determine the optimal replenishment quantity Q as well as the reorder point R following the (Q,R) model.
* _**"RL_NN.py"**_: The _**"RL_NN.py"**_ script is used to determine the actions using the trained neural networks. For each POMDP approximator the trained network is loaded and the actions are determined.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Benedict Rau - benedict.rau@gmx.de

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/benedictrau/Master-Thesis/blob/main/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
 
