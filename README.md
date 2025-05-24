Here is the relevant dataset and open-source code for the article titled "Atten-TPL: A Novel TPL Recommendation Model Based on Attention Mechanism"

&#x20;

*   The “Atten-TPL” folder includes datasets used for experiment and the code of our proposed Atten-TPL model.

*   The “Baseline” folder contains the codes of all baseline models, including: CrossRec, LibSeek, GRec, HCF.

## Atten-TPL

### Usage

#### **Step 1. Data downloading**

Due to GitHub size limitations, please download the data from the link below:

📦 [Download ](https://drive.google.com/your-link-here)`data.zip` (Google Drive)

#### **Step 2. Unzip the data**

After downloading, unzip `data.zip`. Then move the extracted `data` folder into the root directory of the `Atten_TPL` project.

#### **Step 3. Generating training data and testing data**

The raw data is located in the `data` folder.\
To prepare the data for model training and evaluation, run the following command:

    python load_data.py 

#### **Step 4. Model training**

Simply start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **train\_model.py**:

    python train_model.py --train_dataset training_0.json 

File name of the training dataset. `--train_dataset <filename>`

Once the program execution is complete, it will generate a folder named **"model\_Atten\_TPL"**, where the trained model is stored.

#### **Step 5. Model testing**

Start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **test\_model.py**:

    python test_model.py --test_dataset testing_0_3.json

File name of the testing dataset. `--test_dataset <filename>`

Once the program execution is complete, it will generate a folder named "output", where the recommendation results is stored.

#### Step 6. Metrics evaluation

Start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **metrics.py**:

    python metrics_single_file.py --fold 0 --rm 1

Then, you may receive the results as follows:

    Top-rm  => MP: 0.673055  MR: 0.673055  MF: 0.673055  MAP: 0.673055  COV: 0.364351
    Top-2rm  => MP: 0.384110  MR: 0.768219  MF: 0.512146  MAP: 0.720637  COV: 0.424640

### Environment Settup

Our code has been tested under Python 3.12.3. The experiment was conducted via PyTorch, and thus the following packages are required:

    numpy==1.26.4
    torch==2.7.0+cu126
    tqdm==4.66.4

Updated version of each package is acceptable.

### Description of Essential Folders and Files

| Name                 | Type   | Description                                                                                                                                                                                                                                                                         |
| :------------------- | :----- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data                 | Folder | Data files required for the experiment. Specifically: **relation\_shuffle.json** contains the apps used in the experiment and their corresponding TPLs. **app\_info.json** contains the description texts for each app, which are crawled from app stores or third-party websites.  |
| experiment data      | Folder | Experiment data of attention scores comparison and function extraction                                                                                                                                                                                                              |
| train\_model.py      | File   | Model training python file of AttenTPL                                                                                                                                                                                                                                              |
| test\_model.py       | File   | Model testing python file of AttenTPL                                                                                                                                                                                                                                               |
| model\_Atten\_TPL.py | File   | Model modules of AttenTPL                                                                                                                                                                                                                                                           |
| utility              | Folder | Tools and essential libraries used by AttenTPL                                                                                                                                                                                                                                      |

### Other Important module of AttenTPL

#### Obtaining app functions from description

Users could obtain app functions from description by executing the **obtain\_function.py**. The program utilizes GPT-3.5 Turbo model to analyze the description text of the app and extract its functions.

1.  **Obtain an API Key**

    Before using this program, you need an OpenAI API key. If you don't have one yet, you can register and obtain it from the [OpenAI website](https://openai.com/).

2.  **Configure the API Key**

    In the code file **obtain\_function.py**, locate `openai.api_key = ""` and insert your API key within the quotation marks.

3.  **Execute the Program**

    Locate `description = ""` on line 32 and insert the app description within the quotation marks. Then, execute the program to obtain app functions.

#### Vectoring app function

Users could represents the function list of app product with a vector by executing the **vectoring\_function.py**. The program represents the function list with a vector by utilizing the Sentence-Bert model.

1.  **Install dependency**

    Ensure you have the necessary dependency installed. You can install it using the pip command: `pip install -U Sentence-transformers`

2.  **Execute the Program**

    Provide the list of function sentences in the `functions` variable. Then, execute the program to obtain obtain feature vectors for app functions.

## Baseline method

### CrossRec

This is a method that employs the CF-based technique to recommend TPLs for target open-source projects.

To provide further insight into this method, you can read the original paper:

> Nguyen, Phuong T., et al. "CrossRec: Supporting software developers by recommending third-party libraries." *Journal of Systems and Software* 161 (2020): 110460.

### LibSeek

This method employs a matrix factorization model to find potentially useful TPLs for apps.

To provide further insight into this method, you can read the original paper:

> He, Qiang, et al. "Diversified third-party library prediction for mobile app development." *IEEE Transactions on Software Engineering* 48.1 (2020): 150-165.

### GRec

This is the first GNN-based TPL recommendation method, and it models the app-TPL interactions as a graph and utilizes GNN to process the graph to provide potentially available TPLs for apps.

To provide further insight into this method, you can read the original paper:

> Li, B., He, Q., Chen, F., Xia, X., Li, L., Grundy, J.,and Yang, Y., 2021. Embedding App-Library Graph for Neural Third Party Library Recommendation. In proceddings of FSE 2021. DOI:10.1145/3468264.3468552

### HCF

This is the state-of-the-art GNN-based TPL recommendation method, and it can return more accurate and diverse recommendation results than other GNN-based methods by modeling apps and TPLs as two hypergraphs and using hypergraph neural network (HGNN) to process these hypergraphs.

To provide further insight into this method, you can read the original paper:

> Chen, Lianrong, et al. "High-Order Collaborative Filtering for Third-Party Library Recommendation." *2023 IEEE International Conference on Web Services (ICWS)*. IEEE, 2023.

### PyRec

This is a state-of-the-art model of TPL recommendation that embeds projects, TPLs, contextual

information, and relations between them into a knowledge graph. It then uses the GNN with attention

mechanisms to capture useful information from the graph to make TPL recommendations. An important

innovation in this model is the use of the attention mechanisms to automatically determine the usefulness

of different types of relations.

To provide further insight into this method, you can read the original paper:

> Bo Li, et al. 2024. Neural Library Recommendation by Embedding Project-Library Knowledge Graph. \*IEEE Transactions on Software Engineering \*50, 6 (2024), 1620–1638.

