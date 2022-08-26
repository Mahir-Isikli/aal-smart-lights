import { Schema, model } from 'mongoose';
import LampGroup from "./LampGroup.interface";

const schema = new Schema<LampGroup>({
    id: { type: String, required: true, unique: true },
    name: String,
    location: Number,
    category: Number,
    color: String,
    intensity: Number
})

export default model("LampGroup", schema)
