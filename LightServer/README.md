<h1>Light Server</h1>

<h2>Docker</h2>
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

<h2>Setup</h2>
- Install node packages via "npm i"
- install mongodb: https://medium.com/create-a-clocking-in-system-on-react/creating-a-local-mongodb-database-and-insert-a-document-c6a4a2102a22
- will provide our mongoDB setup via Docker container soon 
