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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In the context of the master thesis, the use of reinforcement learning in inventory management in the retail sector was evaluated. <br>
This README file is intended to give an overview of the scripts used in the project and to provide the reader with instructions on how to execute the scripts independently.
`project_description`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Required Packages

The following libraries with their respective versions are used in this project:

* [Pytorch](https://pytorch.org): v1.12.1
* [SimPy](https://simpy.readthedocs.io/en/latest/index.html) v4.0.1
* [Numpy](https://simpy.readthedocs.io/en/latest/index.html) v1.22.4
* [Pandas](https://simpy.readthedocs.io/en/latest/index.html) v1.3.2
* [Scikit-Learn](https://simpy.readthedocs.io/en/latest/index.html) v1.2.0
* [Joblib](https://simpy.readthedocs.io/en/latest/index.html) v1.2.0
* [matplotlib](https://simpy.readthedocs.io/en/latest/index.html) v3.4.1
* [xgboost](https://simpy.readthedocs.io/en/latest/index.html) v1.6.2


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
   pip install -package name-
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Overview -->
## Overview of the Python scripts used

In the following, an overview of the scripts used in the project is given based on the folder structure. 

### Predict Stock

#### Generate the data to train the classification models
This folder contains all scripts and files used to generate the belief state. 
The file _**"Main_predict_stock_createCSV.py"**_ was used to generate the data for training the classification models.
This script interacts with the reduced simulation model of the inventory system that can be found under the file name _**"Env_predict_stock.py"**_. <br>
The generated .csv files are stored in the folder _**"Data"**_.
<br><br>

#### Classification models
In total, the following three classification algorithms were tested as part of the master's thesis:
* Random forest (_**RF.py**_)
* Support vector machine (_**SVM.py**_)
* XGBoost (_**XGB.py**_)
<br>

The scripts each contain the following four functions:
* train(): This function is used to train and save the model.
* HP_opt(): This function performs random search.
* predict(): This function returns the most probable class based on a trained model.
* class_probability(): This function returns the probabilities of each class based on a trained model.

The trained models are stored in the _**"Results"**_ folder and are also called up from there for prediction.
<br><br>

### Reinforcement_Learning

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Benedict Rau - benedict.rau@email_client.com

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
 