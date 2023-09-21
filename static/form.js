const { useState } = React;

function Form(props){
    const [hashtag, setHashtag] = useState('');

    return <div className="form-control grid  place-items-center my-6">
    <label className="label">
      <span className="label-text">Enter any hashtag you want to search?</span>
    </label>
    <input type="text" placeholder="Type here" className="input input-bordered w-full max-w-xs" 
        onChange={(event) => {setHashtag(event.target.value)}} value = {hashtag}
    />
    <button className="btn btn-outline my-1 btn-sm"
        onClick={ async () => {
            console.log(hashtag)
            const res = await fetch(encodeURI(`http://localhost:8000/trend/${hashtag}`));
            const data = await res.json()
            props.setData(data)
            console.log(data)
        }}>Find Sentiment</button>
  </div>
}