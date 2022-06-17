import { Schema, model } from 'mongoose';
import LightEvent from "./LightEvent.interface";

const schema = new Schema<LightEvent>({
    id: {type: String, required: true, unique: true},
    scenes: Array
})

export default model("LightEvent", schema)
