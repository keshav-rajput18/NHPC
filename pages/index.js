import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import axios from "axios";

export default function Home() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [uploadedText, setUploadedText] = useState("");

  const handleUpload = async () => {
    if (!uploadedText) return;
    await axios.post("http://localhost:8000/upload", { text: uploadedText });
    alert("Text uploaded successfully!");
  };

  const handleSearch = async () => {
    if (!query) return;
    const res = await axios.post("http://localhost:8000/search", { query });
    setResponse(res.data);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">AI-Powered RAG Search</h1>
      
      <Card className="mb-4">
        <CardContent className="p-4">
          <Input
            placeholder="Enter text to upload"
            value={uploadedText}
            onChange={(e) => setUploadedText(e.target.value)}
          />
          <Button className="mt-2" onClick={handleUpload}>Upload</Button>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <Input
            placeholder="Enter query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button className="mt-2" onClick={handleSearch}>Search</Button>
        </CardContent>
      </Card>
      
      {response && (
        <Card className="mt-4">
          <CardContent className="p-4">
            <h2 className="font-semibold">Response:</h2>
            <p>{response}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
