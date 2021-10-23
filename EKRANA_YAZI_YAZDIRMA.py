# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 11:56:57 2021

@author: ahmthaydrornk
"""

import numpy as np
import cv2

"""
USB kameradan görüntü almak için VideoCapture(1) kullanılır.
"""
  
KAMERA = cv2.VideoCapture(1)

KOORDINATLAR = []

"""
Kamera'dan sürekli olarak görüntü okumak için sonsuz bir döngü oluşturulur.
"""

while True:

    """ 
    .read() fonksiyonu ile KAMERA'dan anlık bir görüntü okunur ve
    bu görüntü OKUNAN_RENKLI_GORUNTU değişkenine aktarılır. Sorunsuz bir 
    şekilde okunursa OKUNDU_BILGISI True olarak dönecektir.
    """

    OKUNDU_BILGISI, OKUNAN_RENKLI_GORUNTU = KAMERA.read()

    """
    .flip() fonksiyonu ile okunan görüntüyü aynalıyoruz.
    """
  
    OKUNAN_AYNALANMIS_RENKLI_GORUNTU = cv2.flip(OKUNAN_RENKLI_GORUNTU, 1)

    """
    Opencv görüntüleri BGR (mavi, yeşil ve kırmızı) olarak okur, renk bulma
    işlemleri için HSV (renk tonu, doyum, değer) uzayı daha uygun olduğundan 
    BGR'den HSV'ye çevrilir.
    """

    OKUNAN_AYNALANMIS_HSV_GORUNTU = cv2.cvtColor(OKUNAN_AYNALANMIS_RENKLI_GORUNTU, 
                                                 cv2.COLOR_BGR2HSV)

    """
    Aranan renge ait alt ve üst eşik değerleri tanımlanır.
    """
  
    MAVI_ALT_ESIK = np.array([90, 100, 150], np.uint8)
    MAVI_UST_ESIK = np.array([120, 255, 255], np.uint8)

    """
    HSV görüntü, belirlenen aralıkta kalacak şekilde maskelenir. Böylece sadece
    ilgili renge ait alanlar resim üzerinde kalmış olur.
    """

    MAVI_MASKE = cv2.inRange(OKUNAN_AYNALANMIS_HSV_GORUNTU, MAVI_ALT_ESIK, 
                             MAVI_UST_ESIK)

    """
    Maskelenmiş görüntü içerisindeki kenarlar belirlenir.
    """
    
    KENARLAR, _ = cv2.findContours(MAVI_MASKE,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    """
    Renge ait kenarlar belirlendikten sonra bu kenarların alanı belirlenir.
    Bu alan belli bir değerin üzerinde ise kenarlara en uygun dikdörtgen
    oluşturularak koordinatları x, y, w, h olarak tutulur.
    """
      
    for pic, KENAR in enumerate(KENARLAR):
        
        RENK_ALANI = cv2.contourArea(KENAR)
        
        if(RENK_ALANI > 30):
            
            x, y, w, h = cv2.boundingRect(KENAR)
            
            OKUNAN_AYNALANMIS_RENKLI_GORUNTU = cv2.rectangle(OKUNAN_AYNALANMIS_RENKLI_GORUNTU, 
                                                           (x, y), 
                                                           (x + w, y + h), 
                                                           (0, 0, 255), 2)

            """
            Resim üzerine yazı yazdırmak için koordinatlar bir listeye eklenir
            ve çizdirme işlemi yapılır. 
            """
 
            KOORDINATLAR.append((x,y))
           
            for KOORDINAT in KOORDINATLAR:
                cv2.circle(OKUNAN_AYNALANMIS_RENKLI_GORUNTU, tuple(KOORDINAT), 
                           radius=3, color=(255, 255, 255), thickness=-1)
  
                       
    cv2.imshow("RENK TESPITI VE YAZI", OKUNAN_AYNALANMIS_RENKLI_GORUNTU)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        
        KAMERA.release()
        
        cv2.destroyAllWindows()
        
        break


