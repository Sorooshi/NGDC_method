# Nesterov-Momentum Gradient Descent Clustering


**Abstract.**

Enhancing the effectiveness of clustering methods has always been of great interest. Therefore, inspired by the success 
story of the gradient descent approach in supervised learning in the current research, we proposed an effective 
clustering method using the gradient descent approach. As a supplementary device for further improvements, 
we implemented our proposed method using an automatic differentiation library to facilitate the users in applying any 
differentiable distance functions. We empirically validated and compared the performance of our proposed method with 
four popular and effective clustering methods from the literature on 11 real-world and 720 synthetic datasets. Our
experiments proved that our proposed method is valid, and in the majority of the cases, it is more effective than the 
competitors.



To use the NGDC, first clone to this repository; next, in the directory where the main.py is located, run the following command:

    python main.py --algorithm_name="gdcm_f" --init="random" --max_iter=10  --run=1 --data_name="iris" --update_rule="ngdc" --mu_1=0.45 --verbose=1 --step_size=0.001 --n_init=10


Currently only pandas dataframes are accepted and the must be located in ./Datasets/F/ directory, unless a user modifies it.



For citation:

    @Article{math11122617,
    AUTHOR = {Shalileh, Soroosh},
    TITLE = {An Effective Partitional Crisp Clustering Method Using Gradient Descent Approach},
    JOURNAL = {Mathematics},
    VOLUME = {11},
    YEAR = {2023},
    NUMBER = {12},
    ARTICLE-NUMBER = {2617},
    URL = {https://www.mdpi.com/2227-7390/11/12/2617},
    ISSN = {2227-7390},
    DOI = {10.3390/math11122617}
    }




