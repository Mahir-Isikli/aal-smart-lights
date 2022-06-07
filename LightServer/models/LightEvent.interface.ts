import { Schema, model } from 'mongoose';

interface LightEvent {
    id: String,
    scenes: Array<String>
}

const schema = new Schema<LightEvent>({
    id: {type: String, required: true, unique: true},
    scenes: Array
})

export default model("LightEvent", schema)
