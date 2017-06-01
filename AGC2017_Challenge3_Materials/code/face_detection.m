function [faces] = face_detection(image)
    det_faces = MyFaceDetectionFunction(image);        
    biggest = zeros(2,2);
    faces=zeros(2,4);
    for z=1:size(det_faces,1)
        biggest = sortrows(biggest);
        width = abs(det_faces(z,1)-det_faces(z,3));
        height = abs(det_faces(z,2)-det_faces(z,4));
        if width*height > biggest(1,1)
            biggest(1,2) = z;
            biggest(1,1) = width*height;
        end
    end
    biggest(~any(biggest,2),:)=[]; %Delete rows with only zeros
    for b=1:size(biggest,1)
        faces(b,:) = det_faces(biggest(b,2),:);
    end
    faces(~any(faces,2),:)=[];
   
end

function [coordinates] =  MyFaceDetectionFunction(image)
    coordinates = [];
    actual_faces = 0;
    models = {'FrontalFaceCART', 'FrontalFaceLBP', 'ProfileFace'};
    m=1;
    while actual_faces==0 && m<=size(models,2)
        faceDetector = vision.CascadeObjectDetector(models{m});
        faceBbox = step(faceDetector, image);
        for i=1:size(faceBbox,1)
            noseDetector = vision.CascadeObjectDetector('nose');                 
            noseDetector.UseROI = true;
            noseBox = step(noseDetector, image, faceBbox(i,:));
            
            x = faceBbox(i,1);
            y = faceBbox(i,2);
            width = x + faceBbox(i,3);
            height = y + faceBbox(i,4);

            if size(noseBox, 1) > 0
                actual_faces = actual_faces + 1;
                coordinates(actual_faces,:) = [x, y, width, height];
            else
                eyeDetector = vision.CascadeObjectDetector('RightEye');
                eyeDetector.UseROI = true;
                eyeBbox = step(eyeDetector, image,  faceBbox(i,:));
                if size(eyeBbox, 1) > 0
                    actual_faces = actual_faces + 1;
                    coordinates(actual_faces,:) = [x, y, width, height];
                end
            end
        end
        m=m+1;
    end
end