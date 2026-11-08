# Drawing data placeholder (raw data is never committed)

Place spiral images here after obtaining HandPD and NewHandPD from their owners (UNESP, Brazil; check each
dataset's terms of use):

    handpd/{healthy,parkinson}/*.png
    newhandpd/{healthy,parkinson}/*.png
    subjects.csv                      # filename,subject_id,source[,label]  (source = handpd | newhandpd)
    newhandpd/signals/<stem>.csv      # optional pen-sensor signals

Without `subjects.csv`, each image is its own "subject" and subject-wise splits may leak people.
If this folder holds no images, `make train` runs the drawing pipeline on SYNTHETIC spirals as a smoke test.
