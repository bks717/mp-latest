FROM node:22-slim

WORKDIR /app

COPY package*.json ./

RUN apt-get update && apt-get install -y python3 build-essential && rm -rf /var/lib/apt/lists/*

RUN npm install

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
