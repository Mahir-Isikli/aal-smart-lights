import { Schema, model } from 'mongoose';

interface Lamp {
    id: String,
    type: String,
    hasColor: boolean,
    hasIntensity: boolean
}

const schema = new Schema<Lamp>({
    id: {type: String, required: true, unique: true},
    type: {type: String, required: true},
    hasColor: Boolean,
    hasIntensity: Boolean
})

// model("Lamp", schema)
export default model("Lamp", schema)

