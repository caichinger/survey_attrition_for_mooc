# survey attrition

## Description
This repo contains the code and notebooks concerning survey attrition inside the [Austrian National Election Study](https://autnes.at/) (AUTNES). This code accompanies the corresponding lecture within the framework of the MOOC titled ...


## Get the data
In order to run the code one has to obtain the data from [AUSSDA dataverse](https://data.aussda.at/) first.
To be able to connect with the Dataverse API you will need an account with the AUSSDA dataverse. You can sign up through the SSO of your institution or using your email. After you have generated your account you'll need the `DATAVERSE_KEY`, which is the API Token that connects your API request with your registered dataverse account. You can obtain the API Token by logging into your account, clicking on your name in the top left, and selecting API Token as can be seen in the picture below.

![Finding API Token](api1.png)

After clicking on API Token you will be taken to a page (image below) where you can generate a 37 digit Token that is valid for one year. Under no circumstances should you share this token with anyone. Treat it like your username/password combination and make sure it is never included in code you share with others or push into a publicly available repository.

![Generating API Token](api2.png)

Once you have the API Token you can paste it into the `get_data.py` inside the `data/` folder and run it.



## Austrian National Election Study 2017

The AUTNES panel surveys for the most recent Austrian election in 2017 were collected in two different ways. There exists an online panel study with six waves (4 pre- and 2 post-election) as well as a multi-mode study with phone and online modes (2 pre- and 1 post-election waves each). The modes, sample sizes, and survey times are very well explained [here](https://autnes.at/en/autnes-data/general-election-2017/). In this repository we focus on the online panel.


