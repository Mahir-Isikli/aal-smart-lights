import { Schema, model } from 'mongoose';
import LightScene from "./Scene.interface";

const schema = new Schema<LightScene>({
    id: {type: String, required: true, unique: true},
    lampConfigs: Array
})

export default model("LightScene", schema)
