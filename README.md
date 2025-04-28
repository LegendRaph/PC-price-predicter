# PC-price-predicter
This project builds a simple machine learning model to predict the price of a laptop based on its key features like:
1. Amount of RAM
2. Storage (SSD or HDD)
3. Company (brand)
4. Product model
5. Screen resolution

You just type basic details about a laptop (for example: "8GB RAM, 512GB SSD, Dell, XPS 13, Full HD 1920x1080") — and the model will estimate its price in Euros (€).

How it works (simple explanation):
1. The laptop dataset is cleaned and prepared (fixing things like RAM and storage formats).
2. Important features are picked out and used to train a machine learning model (a Decision Tree).
3. When you input a laptop's details, the model processes your text and tries to "understand" it even if you don't type 
   everything perfectly (using fuzzy matching).
4. The model then predicts a price based on the specs you gave.

