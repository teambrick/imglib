import pytesseract
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import math
import autocorrect

spell = autocorrect.Speller(fast=True)


# helper for parsing pytesseract output {{{
def parse_data(data):
    out = []
    for i in data.split("\n")[1:]:
        p = [j for j in i.split() if j != ""]
        if len(p) != 12: #no text
            continue
        [level, pg_num, blk_num, par_num, line_num, word_num, left, top, width, height, conf, txt] = p
        out.append({"txt":spell(txt), "conf": conf})
    return out
# }}}

def ocr(image, guess_clusters: bool = False, train_size: int = 60, cls_low: int=3, cls_hi: int=7):
    # load image
    img_rgb = np.array(image)

    # LAB supremacy
    img = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)

    img = cv.GaussianBlur(img, (3, 3), 3.0)

# shrink for faster k-means-clustering {{{
    # if the maxdim is newsize, what's the mindim value for shrunk image?
    newsize = train_size

    sf = (newsize/max(img.shape))
    newshape = (math.ceil(sf*img.shape[0]), math.ceil(sf*img.shape[1]))

    img_small = cv.resize(img, newshape)

    img_small = cv.GaussianBlur(img_small, (3, 3), 3.0)
# }}}

    # flatten it
    img_flat = np.reshape(img_small, (-1, 3))

    best_n_clusters = 4

# cluster guessing if it's enabled {{{
    if guess_clusters:
        print(f"guessing clusters from {cls_low} to {cls_hi}")
        sil_score_max = -1 #this is the minimum possible score
        for n_clusters in range(cls_low, cls_hi+1):
            model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init="auto")
            labels = model.fit_predict(img_flat)
            sil_score = silhouette_score(img_flat, labels)
            # print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
            if sil_score > sil_score_max:
                sil_score_max = sil_score
                best_n_clusters = n_clusters

        print("best clusters =", best_n_clusters)

#}}}

    # make + train it
    cluster = KMeans(n_clusters=best_n_clusters, n_init="auto")
    cluster.fit(img_flat)

    imgsplit = np.array([cluster.predict(i) for i in img])

    outs = []

# try it against all the splits {{{
    for i in range(len(cluster.cluster_centers_)):
        imgnew = np.equal(imgsplit, i).astype(np.uint8)
        imgnew = np.multiply(imgnew, 255)
        imgcol = cv.cvtColor(imgnew, cv.COLOR_GRAY2RGB)
        # tidy it up a bit
        imgcol = cv.GaussianBlur(imgcol, (3, 3), 5.0)
        # plt.imshow(imgcol)
        # plt.show()
        outs.append(parse_data(pytesseract.image_to_data(imgcol, config="imglib/tconf")))
# }}}

    # cleanup each guess
    # pytesseract likes to spit out random letters and weird characters, none of which we want
    cleaned = [[{"conf":i["conf"], "txt":i["txt"].strip().lower()} for i in out if all([j.isalpha() for j in i["txt"].strip()]) and len(i["txt"]) > 1] for out in outs]

# score each one {{{
    confs = []
    for out in cleaned:
        if out == []:
            confs.append(0)
            continue
        total = sum(float(i["conf"]) for i in out)
        total /= len(out)
        # add decaying bonus for longer output
        bonus = -math.exp(-len(out)/10)+1
        bonus *= 0.5
        confs.append(bonus+total)
# }}}

    # find the best
    best = max(zip(confs, cleaned), key=lambda x: x[0])
    # print(best)

    guess = [i["txt"] for i in best[1]]
    # print("Best Guess:")
    # print(guess)
    return guess
