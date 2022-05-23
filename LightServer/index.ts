import express, { Express } from 'express';
import dotenv from 'dotenv';
import swaggerUi from 'swagger-ui-express'
import swaggerJsdoc from 'swagger-jsdoc'

import lightRoutes from './routes/lights/toggle'

dotenv.config();

const app: Express = express();
const cors = require('cors')
const port = process.env.PORT || 8080; //;

app.use(cors({ credentials: true, origin: true }))

app.use(express.static("public"));

app.use('/lights', lightRoutes);

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Smart lights',
      version: '1.0.0',
    },
  },
  apis: ['./routes/**/*.ts']
}

const swaggerSpec = swaggerJsdoc(options)
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec))

app.listen(port, () => {
  console.log(`⚡️[server]: Server is running at https://localhost:${port}`);
});
