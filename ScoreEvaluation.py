import pickle 
  
def ScoreEvaluation(pre_status,score,H_count,S_count,angle,pre_angle,body_language_class):
#자세 점수 감점
    if pre_status[0] == 0 and body_language_class.split(' ')[0]=='Bad':
        score[0] -= 1
        H_count[0] += 1
    

    if pre_angle[0] == 0 and angle < 170:
        score[0] -= 1
        S_count[0] += 1
    
    #중복 감점 방지
    if body_language_class.split(' ')[0]=='Normal':
        pre_status[0] = 0
    elif body_language_class.split(' ')[0]=='Bad':
        pre_status[0] = 1

    if angle < 170:
        pre_angle[0] = 1
    else:
        pre_angle[0] = 0