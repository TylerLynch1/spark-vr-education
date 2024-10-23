# Impact of Virtual Reality on Education Analysis

## Project Overview
This project analyzes the impact of Virtual Reality (VR) on education using a dataset obtained from Kaggle. The analysis aims to explore how VR technologies can enhance learning outcomes and student engagement in educational settings. The project utilizes Apache Spark for data preprocessing and statistical analysis, leveraging distributed computing to handle large datasets efficiently.

## Dataset
The dataset used for this analysis is the **Impact of Virtual Reality on Education** dataset, which includes various attributes related to students' experiences and outcomes with VR in educational contexts.

- **Source**: [Kaggle - Impact of Virtual Reality on Education](https://www.kaggle.com/datasets/waqi786/impact-of-virtual-reality-on-education)
- **File Format**: CSV

## Project Structure
spark-vr-education/
│ 
├── data/                   # Folder containing the dataset
│   └── dataset.csv         # The dataset file 
│ 
├── src/                    # Source code for analysis 
│   ├── preprocessing.py     # Data preprocessing scripts 
│   └── analysis.py         # Statistical analysis scripts 
│ 
├── requirements.txt        # Python package dependencies 
├── README.md               # Project documentation 
├── LICENSE                 # License information for the project
└── .gitignore              # Files and directories to ignore

## Installation
To run this project, you need to have the following software installed:

- Python 3.x
- Apache Spark
- Git

### Python Packages
The required Python packages are listed in `requirements.txt`. You can install them using:

```bash
pip install -r requirements.txt

git clone https://github.com/TylerLynch1/spark-vr-education.git
cd spark-vr-education

python src/preprocessing.py

python src/analysis.py
```

Performance Comparison
The project includes a performance comparison of the analysis running on one VM versus two VMs. Performance metrics such as execution time and resource usage will be documented.

The files vr_education.py and vr_education_hadoop.py are the same except for the file path.

Contributions
This project was made be Tyler Lynch for the Fall 2024 - SAT 5165 course. 

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Kaggle for providing the dataset.
Apache Spark community for their documentation and resources.
