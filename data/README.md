# Data

`weekly_demand.csv` is committed - it is small, and it is what the app reads.

The raw source is **not** committed (44 MB). To rebuild everything from scratch:

```bash
curl -L -o data/online_retail_II.zip \
  https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip data/online_retail_II.zip -d data/
python data/prepare_data.py
```

Source: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
UCI Machine Learning Repository. Two years of transaction lines from a UK-based
online retailer, Dec 2009 - Dec 2011.
