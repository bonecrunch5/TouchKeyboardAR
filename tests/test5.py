#!/usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__' :

    # Read source image.
    im_src = cv2.imread('book2_edited.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[111,123], [230,153], [293,402],[130, 323]])

    # Read destination image.
    im_dst = cv2.imread('book2.jpg')
    # Four corners of the book in destination image.
    #pts_dst = np.array([[1,1],[333,1],[333, 444],[1, 444]])
    pts_dst = np.array([[1,1],[333,1],[333, 444],[1, 444]])


    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)
