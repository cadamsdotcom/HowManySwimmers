# HowManySwimmers

Face detection to count how many swimmers in a Salties photo..

I tried SAM (Segment Anything) but that doesn't have a classifier, and I tried YOLO (you only look once) with full body detection, but that barely worked at all...



Model doesn't do so well with the really epic pics of hundreds of swimmers, the pics from January came back with 0 faces. So it needs a beefing up for summer. But at least here's a first cut y'all can iterate off of.


Usage:
```
git clone https://github.com/thecadams/HowManySwimmers
cd HowManySwimmers
python -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
flask run --port=8080 # default port 5000 is in use on macOS
```



Made with help from Claude 3.5 Sonnet, plus a sprinkle of salty love 🦀❤️