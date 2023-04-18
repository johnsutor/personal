import { useState } from "react"

export default function VideoSlider(video: string) {
    const [slider_val, setSliderVal] = useState(50)
    return (
        <div className="mx-auto">
            <video width="100%" autoPlay loop muted>
                <source src={video} type="video/mp4" />
            </video>
            <div className="w-full">
                <input type="range" min="1" max="100" value={slider_val} id="myRange" onChange={e => setSliderVal(e.target.value)} />
            </div>
        </div>
    )
}