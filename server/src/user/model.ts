import { IDbUser, RWSModel } from "@rws-framework/server";

class JWTUser implements IDbUser {
    mongoId: string;
    loadDbUser: () => Promise<void>;
    db: RWSModel<any>;
}

export { JWTUser };