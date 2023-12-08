# 🧑‍🍳 L7 - You're still here?

https://github.com/microsoft/chat-copilot

![](./assets/cc_githubrepo.png)

![](./assets/cc_plain.png)

![](./assets/cc_chatgptplugin.gif)

![](./assets/cc_vectordbs.jpg)

![](./assets/cc_settings.png)

## 🎁 The backend server demonstrates how to connect to a variety of resources like auth, vector dbs, telemetry, content safety, PDF import, and even OCR. 

```
//
// # ABBREVIATED Chat Copilot Application Settings
{
  "AIService": {
    "Type": "AzureOpenAI",
    "Endpoint": "", // ignored when AIService is "OpenAI"
     "Key": "",
    "Models": {
      "Completion": "gpt-35-turbo", 
      "Embedding": "text-embedding-ada-002",
      "Planner": "gpt-35-turbo" 
    }
  },
  //
  // Optional Azure Speech service configuration for providing Azure Speech access tokens.
  // - Set the Region to the region of your Azure Speech resource (e.g., "westus").
  // - Set the Key using dotnet's user secrets (see above)
  //     (i.e. dotnet user-secrets set "AzureSpeech:Key" "MY_AZURE_SPEECH_KEY")
  //
  "AzureSpeech": {
    "Region": ""
    // "Key": ""
  },
  //
  // Authorization configuration to gate access to the service.
  // - Supported Types are "None", "ApiKey", or "AzureAd".
  // - Set ApiKey using dotnet's user secrets (see above)
  //     (i.e. dotnet user-secret set "Authorization:ApiKey" "MY_API_KEY")
  //
  "Authorization": {
    "Type": "None",
    "ApiKey": "",
    "AzureAd": {
      "Instance": "https://login.microsoftonline.com/",
      "TenantId": "",
      "ClientId": "",
      "Audience": "",
      "Scopes": "access_as_user" // Scopes that the client app requires to access the API
    }
  },
  //
  // Chat stores are used for storing chat sessions and messages.
  // - Supported Types are "volatile", "filesystem", or "cosmos".
  // - Set "ChatStore:Cosmos:ConnectionString" using dotnet's user secrets (see above)
  //     (i.e. dotnet user-secrets set "ChatStore:Cosmos:ConnectionString" "MY_COSMOS_CONNSTRING")
  //
  "ChatStore": {
    "Type": "volatile",
    "Filesystem": {
      "FilePath": "./data/chatstore.json"
    },
    "Cosmos": {
      "Database": "CopilotChat",
      "ChatSessionsContainer": "chatsessions",
      "ChatMessagesContainer": "chatmessages",
      "ChatMemorySourcesContainer": "chatmemorysources",
      "ChatParticipantsContainer": "chatparticipants"
      // "ConnectionString": // dotnet user-secrets set "ChatStore:Cosmos:ConnectionString" "MY_COSMOS_CONNECTION_STRING"
    }
  },
  //
  // Memory stores are used for storing new memories and retrieving semantically similar memories.
  "MemoryStore": {
    "Type": "volatile",
    "Qdrant": {
      "Host": "http://localhost",
      "Port": "6333",
      "VectorSize": 1536
      // "Key":  ""
    },
    "AzureCognitiveSearch": {
      "Endpoint": ""
      // "Key": ""
    },
    "Chroma": {
      "Host": "http://localhost",
      "Port": "8000"
    },
    "Postgres": {
      "VectorSize": 1536
      // "ConnectionString": // dotnet user-secrets set "MemoryStore:Postgres:ConnectionString" "MY_POSTGRES_CONNECTION_STRING"
    }
  },
  //
  // Document import configuration
  //
  "DocumentMemory": {
    "GlobalDocumentCollectionName": "global-documents",
    "ChatDocumentCollectionNamePrefix": "chat-documents-",
    "DocumentLineSplitMaxTokens": 72,
    "DocumentChunkMaxTokens": 512,
    "FileSizeLimit": 4000000,
    "FileCountLimit": 10
  },
  //
  // OCR support is used for allowing end users to upload images containing text in addition to text based documents.
  //  https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/quickstarts-sdk/client-library?tabs=windows%2Cvisual-studio&pivots=programming-language-csharp#optical-character-recognition-ocr-with-computer-vision-api-using-c
  //
  "OcrSupport": {
    "Type": "none",
    "Tesseract": {
      "Language": "eng",
      "FilePath": "./data"
    },
    "AzureFormRecognizer": {
      "Endpoint": ""
      // "Key": "",
    }
  },
  //
  // Application Insights configuration
  // - Set "APPLICATIONINSIGHTS_CONNECTION_STRING" using dotnet's user secrets (see above)
  //     (i.e. dotnet user-secrets set "APPLICATIONINSIGHTS_CONNECTION_STRING" "MY_APPINS_CONNSTRING")
  //
  "APPLICATIONINSIGHTS_CONNECTION_STRING": null
}
```


```python

```
