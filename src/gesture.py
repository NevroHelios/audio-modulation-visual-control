import mediapipe as mp
import cv2
import numpy as np
from src.shared_state import shared_state, lock
from src.config import CFG

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint_list = [[8, 9, 16]] # , [4, 9, 12], [4, 3, 2]




def draw_finger_angles(image, results, joint_list):
    for hand in results.multi_hand_landmarks:
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
                a[1] - b[1], a[0] - b[0]
            )
            angle = np.abs(radians * 180.0 / np.pi)

            cv2.putText(
                image,
                str(int(angle)),
                tuple(
                    np.multiply(b, [CFG.WINDOW_WIDTH, CFG.WINDOW_HEIGHT]).astype(int)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )


def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = f"{label} {score:.2f}"

            cords = tuple(
                np.multiply(
                    np.array(
                        [
                            hand.landmark[mp_hands.HandLandmark.WRIST].x,
                            hand.landmark[mp_hands.HandLandmark.WRIST].y,
                        ]
                    ),
                    np.array([CFG.WINDOW_WIDTH, CFG.WINDOW_HEIGHT]),
                ).astype(int)
            )

            output = (text, cords)

    return output


def hand_tracking():


    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.WINDOW_HEIGHT)

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # print(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                planes = []
                for num, hand in enumerate(results.multi_hand_landmarks):
                    index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # 8
                    thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]  # 4

                    x8, y8 = (
                        int(index_tip.x * CFG.WINDOW_WIDTH),
                        int(index_tip.y * CFG.WINDOW_HEIGHT),
                    )
                    x4, y4 = (
                        int(thumb_tip.x * CFG.WINDOW_WIDTH),
                        int(thumb_tip.y * CFG.WINDOW_HEIGHT),
                    )
                    cv2.circle(frame, (x8, y8), 5, (221, 235, 157), -1)
                    cv2.circle(frame, (x4, y4), 5, (160, 200, 2120), -1)
                    cv2.line(frame, (x8, y8), (x4, y4), (255, 253, 246), 2)

                    label = get_label(num, hand, results)
                    if label:
                        text, cords = label
                        cv2.putText(
                            frame,
                            text,
                            cords,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

                        planes.append(
                            {
                                "text": text,
                                "plane_height": int(
                                    -(index_tip.y - thumb_tip.y) * CFG.WINDOW_HEIGHT
                                ),
                                "cords": (x8, y8, x4, y4),
                            }
                        )

                draw_finger_angles(frame, results, joint_list)
                if len(planes) == 2:
                    # Calculate midpoints for sorting
                    mid_x0 = (planes[0]["cords"][0] + planes[0]["cords"][2]) / 2
                    mid_x1 = (planes[1]["cords"][0] + planes[1]["cords"][2]) / 2

                    # Sort based on midpoint x-coordinate
                    if mid_x0 <= mid_x1:
                        left_hand_plane = planes[0]
                        right_hand_plane = planes[1]
                    else:
                        left_hand_plane = planes[1]
                        right_hand_plane = planes[0]

                    l_x1, l_y1 = left_hand_plane["cords"][0], left_hand_plane["cords"][1]
                    l_x2, l_y2 = left_hand_plane["cords"][2], left_hand_plane["cords"][3]
                    r_x1, r_y1 = right_hand_plane["cords"][0], right_hand_plane["cords"][1]
                    r_x2, r_y2 = right_hand_plane["cords"][2], right_hand_plane["cords"][3]


                    # cv2.line(frame, (l_x2, l_y2), (r_x2, r_y2), (255, 253, 246), 2)
                    # cv2.line(frame, (l_x1, l_y1), (r_x1, r_y1), (255, 253, 246), 2)
                    right_plane_length = right_hand_plane["plane_height"]
                    left_plane_length = left_hand_plane["plane_height"]
                    with lock:
                        shared_state['max_plane_height'] = max(left_plane_length, right_plane_length)

                    # distance_between_planes = abs(
                    #     np.linalg.norm(
                    #         np.array([l_x1, l_y1]) - np.array([r_x1, r_y1])
                    #     )
                    # )
                    new_volume = round(right_plane_length / CFG.WINDOW_HEIGHT * 1.2, 2)
                    with lock:
                        shared_state["volume"] = new_volume

                    spectrum = shared_state.get("spectrum", None)

                    for i in range(CFG.BINS):
                        t = i / CFG.BINS
                        btm_x = int((1 - t) * l_x2 + t * r_x2)
                        btm_y = int((1 - t) * l_y2 + t * r_y2)
                        top_x = int((1 - t) * l_x1 + t * r_x1)
                        top_y = int((1 - t) * l_y1 + t * r_y1)

                        if spectrum is not None:
                            x = int((1 - t) * l_x1 + t * r_x1) # stabiled one
                            bar_height = int(spectrum[i])

                            gap_distance = (top_y - btm_y - bar_height) // 2

                            
                            y1 = int(btm_y +  gap_distance) 
                            y2 = int(btm_y + bar_height + gap_distance) 
                            cv2.line(
                                frame,
                                (x, y1),
                                (x, y2),
                                (255, 255, 255),
                                2,
                            )
                        else:

                            # cv2.circle(frame, (btm_x, btm_y), 5, (255, 253, 246), -1)
                            # cv2.circle(frame, (top_x, top_y), 5, (255, 253, 246), -1)
                            cv2.line(frame, (btm_x, btm_y), (top_x, top_y), (255, 253, 246), 2)
                
            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
