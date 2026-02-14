FROM node:20-slim

WORKDIR /app

COPY website/server/package.json website/server/package-lock.json ./
RUN npm ci --omit=dev

COPY website/server ./

ENV NODE_ENV=production
ENV PORT=3001
ENV DATA_DIR=/data
ENV SQLITE_PATH=/data/synthgen.sqlite
ENV OUTPUTS_DIR=/data/outputs

EXPOSE 3001

CMD ["npm", "start"]
