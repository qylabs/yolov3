from os import path
import cv2


def extract_img(video_path,target_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Failed to open {video_path}'
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
    # frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
    n=1
    while cap.isOpened():
        ret, imgs = cap.read()
        if ret is False:
            break
        if n%10==0:
            target_img_path=path.join(target_path,'img_'+str(n)+'.jpg')
            print(target_img_path)
            cv2.imwrite(target_img_path,imgs)
        n+=1
    
    cap.release()

if __name__=="__main__":
    video_path='cust_data/3.mp4'
    target_path='cust_data/img_3'
    extract_img(video_path,target_path)

