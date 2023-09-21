function Content(props) {
    if (!props.data.scrapped_tweets) {
        return (
            <div className="flex items-center justify-center ">
                <img src="Social.png"></img>
            </div>
        )
        
    }
    return (
        <>
            <div className="stats shadow w-1/3">
                <div className="stat ">
                    <div className="stat-title">Prediction (Logistic Regression)</div>
                    <div className="stat-value">{props.data.prediction}</div>
                    <div className="stat-desc">{props.data.sentiment.toUpperCase()}</div>
                </div>
            </div>
            {/* <div className="stats shadow mx-30 w-1/3">
                <div className="stat ">
                    <div className="stat-title">Prediction (Zero Shot Learning with OpenAssistant/oasst-sft-6-llama-30b)</div>
                    <div className="stat-value">{props.data.zero_shot.sentiment.toUpperCase() + " SENTIMENT"}</div>
                    <div className="stat-desc">{props.data.zero_shot.summary}</div>
                </div>
            </div> */}
            <div className="flex overflow-scroll .no-scrollbar my-6">
                {
                    props.data.scrapped_tweets.map((d) => {
                        return (
                            <div className="card w-96 bg-blue-100 shadow-xl" key={d}>
                                <div className="card-body">
                                    <p>{d}</p>
                                </div>
                            </div>
                        )
                    })
                }
            </div>
        </>
    )
}