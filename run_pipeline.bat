@echo off
call venv\Scripts\activate
python src/task2/pipeline.py --text "There is a cow in the picture" --image data/animals10_demo/cow_1.jpg --classes cat dog cow sheep horse elephant bear zebra giraffe monkey
pause
