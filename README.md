# BatchLegal
## Analyzing EU regulation in detail using NLP
- BatchLegal uses data from the EUR-Lex WebServices tool to perform topic analysis on published EU regulation
- It aims to provide a more granular insight into EU Regulation and what it concerns
- We use BERTopic (https://github.com/MaartenGr/BERTopic) for Transformers-based topic modeling on the EU regulation at a sub-directory level
- The output is visualised in Streamlit and deployed on Heroku at: https://batchlegal.herokuapp.com/

### Data Source
This Python script queries the EURLexWebservice. To use it yourself, you need to request a username and password. This can be done here:
https://eur-lex.europa.eu/content/help/data-reuse/webservice.html

This package uses a two-step process to obtain the data from the EUR-Lex webservice. First, we use the EURLex Web Service to obtain MetaData and *Cellar references*.

Cellar is the repository of the content for documents. More information: https://eur-lex.europa.eu/content/help/data-reuse/reuse-contents-eurlex-details.html

In "scraper.py" we use the Cellar references to attach the text content to each document. In "api.py" we fetch the metadata.
