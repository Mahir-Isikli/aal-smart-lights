How to pull docker container (arm architecture)
- run: docker pull paulfischer/lightserver:arm

How to Build Docker container
- open terminal 
- go to lightserver root dir 
- run: docker build -t lightserver .

How to Run Docker container:
- open terminal 
- go to lightserver root dir 
- run: docker run --rm -itd -p 9999:8080 lightserver
- this will open 


Stop Docker container: 
- run: docker ps 
- copy containerID 
- run: docker stop CONTAINERID 
